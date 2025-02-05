import {
  AgentKit,
  CdpWalletProvider,
  wethActionProvider,
  walletActionProvider,
  erc20ActionProvider,
  cdpApiActionProvider,
  cdpWalletActionProvider,
  pythActionProvider,
} from "@coinbase/agentkit";
import { getLangChainTools } from "@coinbase/agentkit-langchain";
import { HumanMessage } from "@langchain/core/messages";
import { MemorySaver } from "@langchain/langgraph";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { ChatOpenAI } from "@langchain/openai";
import * as dotenv from "dotenv";
import * as fs from "fs";
import * as readline from "readline";
import { customActionProvider } from "@coinbase/agentkit";
import { z } from "zod";
import { DallEAPIWrapper } from "@langchain/openai";


dotenv.config();

/**
 * Validates that required environment variables are set.
 */
function validateEnvironment(): void {
  const missingVars: string[] = [];
  const requiredVars = ["OPENAI_API_KEY", "CDP_API_KEY_NAME", "CDP_API_KEY_PRIVATE_KEY"];
  requiredVars.forEach((varName) => {
    if (!process.env[varName]) {
      missingVars.push(varName);
    }
  });
  if (missingVars.length > 0) {
    console.error("Error: Required environment variables are not set");
    missingVars.forEach((varName) => {
      console.error(`${varName}=your_${varName.toLowerCase()}_here`);
    });
    process.exit(1);
  }
  if (!process.env.NETWORK_ID) {
    console.warn("Warning: NETWORK_ID not set, defaulting to base-sepolia testnet");
  }
}

validateEnvironment();

const WALLET_DATA_FILE = "wallet_data.txt";

/**
 * Define a custom action to get Ether balance using the BaseScan API.
 */
const customGetBalance = customActionProvider({
  name: "get_balance",
  description: "Retrieve Ether balance for a given Ethereum address using BaseScan API",
  schema: z.object({
    address: z.string().describe("The Ethereum address to check balance for"),
  }),
  invoke: async (_walletProvider, args: any) => {
    const { address } = args;
    const apiKey = process.env.BASESCAN_API_KEY; // Replace with your actual BaseScan API key
    const url = `https://api.basescan.org/api?module=account&action=balance&address=${address}&tag=latest&apikey=${apiKey}`;
    try {
      const response = await fetch(url);
      const data = await response.json();
      if (data.status !== "1") {
        throw new Error(`Error from BaseScan: ${data.message}`);
      }
      const balanceWei = data.result;
      return `The balance for address ${address} is ${balanceWei} wei.`;
    } catch (error) {
      return `Failed to retrieve balance: ${(error as Error).message}`;
    }
  },
});

const customGetTokenTransfers = customActionProvider({
  name: "get_token_transfers",
  description:
    "Retrieve ERC-20 token transfer events for an address (and optionally filter by a token contract) using the BaseScan API",
  schema: z.object({
    // If checking transfers from an address, specify the address.
    address: z.string().optional().describe("The Ethereum address to check token transfers for"),
    // If filtering by a specific token contract, specify the contract address.
    contractaddress: z.string().optional().describe("Optional token contract address to filter transfers"),
    // Optional paging and filtering parameters:
    page: z.number().optional().default(1).describe("Page number for pagination"),
    offset: z.number().optional().default(100).describe("Number of results per page"),
    startblock: z.number().optional().default(0).describe("Starting block number for search"),
    endblock: z.number().optional().default(27025780).describe("Ending block number for search"),
    sort: z.string().optional().default("asc").describe("Sort order: 'asc' or 'desc'"),
  }),
  invoke: async (_walletProvider, args: any) => {
    // Destructure parameters, using defaults if not provided.
    const {
      address,
      contractaddress,
      page = 1,
      offset = 100,
      startblock = 0,
      endblock = 27025780,
      sort = "asc",
    } = args;
    
    const apiKey = process.env.BASESCAN_API_KEY; 

    // Construct the URL using the provided parameters.
    let url = `https://api.basescan.org/api?module=account&action=tokentx`;
    if (address) {
      url += `&address=${address}`;
    }
    if (contractaddress) {
      url += `&contractaddress=${contractaddress}`;
    }
    url += `&page=${page}&offset=${offset}&startblock=${startblock}&endblock=${endblock}&sort=${sort}&apikey=${apiKey}`;

    try {
      const response = await fetch(url);
      const data = await response.json();

      // Check if the API response is successful.
      if (data.status !== "1") {
        throw new Error(`Error from BaseScan: ${data.message}`);
      }
      
      // The result is expected to be an array of token transfer events.
      const transfers = data.result;
      return `Token transfer events: ${JSON.stringify(transfers)}`;
    } catch (error) {
      return `Failed to retrieve token transfers: ${(error as Error).message}`;
    }
  },
});

const nftMinterAction = customActionProvider({
  name: "mint_erc721",
  description: "Mint ERC721 NFT to specified address",
  schema: z.object({
    recipient: z.string().describe("Recipient wallet address"),
    tokenURI: z.string().describe("IPFS URI for NFT metadata"),
    contractAddress: z.string().describe("ERC721 contract address")
  }),
  invoke: async (walletProvider, args) => {
    const erc721ABI = [
      "function safeMint(address to, string memory uri) external"
    ];

    try {
      const signer = await walletProvider.getSigner();
      const contract = new ethers.Contract(args.contractAddress, erc721ABI, signer);
      const tx = await contract.safeMint(args.recipient, args.tokenURI);
      await tx.wait();
      return `NFT minted successfully: ${tx.hash}`;
    } catch (error) {
      return `Minting failed: ${(error as Error).message}`;
    }
  }
});

const dallETool = new DallEAPIWrapper({
  n: 1,
  model: "dall-e-3",
  apiKey: process.env.OPENAI_API_KEY,
});


/**
 * Initialize the agent with CDP AgentKit.
 */
async function initializeAgent() {
  try {
    // Initialize LLM
    const llm = new ChatOpenAI({
      model: "gpt-4o-mini",
    });

    let walletDataStr: string | null = null;
    if (fs.existsSync(WALLET_DATA_FILE)) {
      try {
        walletDataStr = fs.readFileSync(WALLET_DATA_FILE, "utf8");
      } catch (error) {
        console.error("Error reading wallet data:", error);
      }
    }

    const config = {
      apiKeyName: process.env.CDP_API_KEY_NAME,
      apiKeyPrivateKey: process.env.CDP_API_KEY_PRIVATE_KEY?.replace(/\\n/g, "\n"),
      cdpWalletData: walletDataStr || undefined,
      networkId: process.env.NETWORK_ID || "base-sepolia",
    };

    const walletProvider = await CdpWalletProvider.configureWithWallet(config);

    // Create AgentKit instance including the custom action.
    const agentkit = await AgentKit.from({
      walletProvider,
      actionProviders: [
        wethActionProvider(),
        pythActionProvider(),
        walletActionProvider(),
        erc20ActionProvider(),
        cdpApiActionProvider({
          apiKeyName: process.env.CDP_API_KEY_NAME,
          apiKeyPrivateKey: process.env.CDP_API_KEY_PRIVATE_KEY?.replace(/\\n/g, "\n"),
        }),
        cdpWalletActionProvider({
          apiKeyName: process.env.CDP_API_KEY_NAME,
          apiKeyPrivateKey: process.env.CDP_API_KEY_PRIVATE_KEY?.replace(/\\n/g, "\n"),
        }),
        customGetBalance,
        customGetTokenTransfers, // Integrate custom action here.
      ],
    });

    const tools = await getLangChainTools(agentkit);

  
    const allTools = [...tools, dallETool];

    const memory = new MemorySaver();
    const agentConfig = { configurable: { thread_id: "CDP AgentKit Chatbot Example!" } };

    const agent = createReactAgent({
      llm,
      tools: allTools,
      checkpointSaver: memory,
      messageModifier: `
        You are a helpful agent that can interact onchain using the Coinbase Developer Platform AgentKit.
        You are empowered to interact onchain using your tools. If you ever need funds, you can request them 
        from the faucet if you are on network ID 'base-sepolia'. If not, you can provide your wallet details and request 
        funds from the user. Before executing your first action, get the wallet details to see what network 
        you're on. If there is a 5XX (internal) HTTP error code, ask the user to try again later. If someone 
        asks you to do something you can't do with your currently available tools, you must say so, and 
        encourage them to implement it themselves using the CDP SDK + Agentkit. Be concise and helpful with your responses.
      `,
    });

    const exportedWallet = await walletProvider.exportWallet();
    fs.writeFileSync(WALLET_DATA_FILE, JSON.stringify(exportedWallet));

    return { agent, config: agentConfig };
  } catch (error) {
    console.error("Failed to initialize agent:", error);
    throw error;
  }
}

/**
 * Run the agent autonomously at specified intervals.
 *
 * @param agent - The agent executor.
 * @param config - Agent configuration.
 * @param interval - Time interval between actions in seconds.
 */
async function runAutonomousMode(agent: any, config: any, interval = 10) {
  console.log("Starting autonomous mode...");
  while (true) {
    try {
      const thought =
        "Be creative and do something interesting on the blockchain. " +
        "Choose an action or set of actions and execute it that highlights your abilities.";
      const stream = await agent.stream({ messages: [new HumanMessage(thought)] }, config);
      for await (const chunk of stream) {
        if ("agent" in chunk) {
          console.log(chunk.agent.messages[0].content);
        } else if ("tools" in chunk) {
          console.log(chunk.tools.messages[0].content);
        }
        console.log("-------------------");
      }
      await new Promise((resolve) => setTimeout(resolve, interval * 1000));
    } catch (error) {
      if (error instanceof Error) {
        console.error("Error:", error.message);
      }
      process.exit(1);
    }
  }
}

/**
 * Run the agent interactively based on user input.
 *
 * @param agent - The agent executor.
 * @param config - Agent configuration.
 */
async function runChatMode(agent: any, config: any) {
  console.log("Starting chat mode... Type 'exit' to end.");
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });
  const question = (prompt: string): Promise<string> =>
    new Promise((resolve) => rl.question(prompt, resolve));
  try {
    while (true) {
      const userInput = await question("\nPrompt: ");
      if (userInput.toLowerCase() === "exit") {
        break;
      }
      const stream = await agent.stream({ messages: [new HumanMessage(userInput)] }, config);
      for await (const chunk of stream) {
        if ("agent" in chunk) {
          console.log(chunk.agent.messages[0].content);
        } else if ("tools" in chunk) {
          console.log(chunk.tools.messages[0].content);
        }
        console.log("-------------------");
      }
    }
  } catch (error) {
    if (error instanceof Error) {
      console.error("Error:", error.message);
    }
    process.exit(1);
  } finally {
    rl.close();
  }
}

/**
 * Prompt the user to choose between chat or autonomous mode.
 *
 * @returns "chat" or "auto" based on user choice.
 */
async function chooseMode(): Promise<"chat" | "auto"> {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });
  const question = (prompt: string): Promise<string> =>
    new Promise((resolve) => rl.question(prompt, resolve));
  while (true) {
    console.log("\nAvailable modes:");
    console.log("1. chat    - Interactive chat mode");
    console.log("2. auto    - Autonomous action mode");
    const choice = (await question("\nChoose a mode (enter number or name): "))
      .toLowerCase()
      .trim();
    if (choice === "1" || choice === "chat") {
      rl.close();
      return "chat";
    } else if (choice === "2" || choice === "auto") {
      rl.close();
      return "auto";
    }
    console.log("Invalid choice. Please try again.");
  }
}

/**
 * Start the chatbot agent.
 */
async function main() {
  try {
    const { agent, config } = await initializeAgent();
    const mode = await chooseMode();
    if (mode === "chat") {
      await runChatMode(agent, config);
    } else {
      await runAutonomousMode(agent, config);
    }
  } catch (error) {
    if (error instanceof Error) {
      console.error("Error:", error.message);
    }
    process.exit(1);
  }
}

if (require.main === module) {
  console.log("Starting Agent...");
  main().catch((error) => {
    console.error("Fatal error:", error);
    process.exit(1);
  });
}
